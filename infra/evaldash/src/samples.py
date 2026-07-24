# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read per-sample contract parquet exports for a run's results directory.

marin.evaluation.samples writes one samples_<task>_<timestamp>.parquet per (sub)task under a run's
results_path, each row an EvalSample. This module discovers those files with fsspec, validates every
row back into the contract model, and returns typed Pydantic responses for the sample browser.
Loaded tables are cached briefly so paging does not re-read object storage on every request.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Generic, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs
from marin.evaluation.samples import SAMPLES_PREFIX, SAMPLES_SUFFIX, EvalSample, primary_metric
from pydantic import BaseModel, ConfigDict
from rigging.filesystem import StoragePath

logger = logging.getLogger(__name__)

CACHE_TTL = 120.0

# A single lazily-loaded artifact (an agentic trajectory, a prediction's exchange) is capped so one
# request cannot pull an unbounded object into memory. Trajectories run tens of KB to a few MB; 16
# MiB leaves generous headroom while still refusing a pathological file. Successful reads are cached
# for the same window as sample tables so re-opening a trajectory does not re-hit object storage.
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024


class SampleTask(BaseModel):
    """One discovered task and the number of parquet shards that contain it."""

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
    """Unpaginated correctness counts for one task."""

    model_config = ConfigDict(frozen=True)

    all: int
    correct: int
    incorrect: int


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
    """One sample-referenced artifact (trajectory/exchange) resolved to text for the browser.

    ``available`` is False -- with a human-readable ``reason`` -- for every non-happy path (the run
    has no results directory, the URI escapes it, the object is missing/unreadable, or it exceeds the
    size cap), mirroring the logs endpoint's reachability degradation rather than raising. ``text`` is
    the decoded object body when available, and ``media_type`` reflects the URI extension so the
    client knows whether to parse JSON.
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


def _sample_task(filename: str) -> str:
    """samples_<task>_<timestamp>.parquet -> <task> (the timestamp has no underscore)."""
    stem = filename[len(SAMPLES_PREFIX) : -len(SAMPLES_SUFFIX)]
    return stem.rsplit("_", 1)[0]


def _discover(results_path: str):
    """Return (fs, {task: [parquet_path, ...]}) for sample files under results_path."""
    fs, root = url_to_fs(results_path)
    by_task: dict[str, list[str]] = {}
    for path in fs.find(root):
        name = path.rsplit("/", 1)[-1]
        if name.startswith(SAMPLES_PREFIX) and name.endswith(SAMPLES_SUFFIX):
            by_task.setdefault(_sample_task(name), []).append(path)
    return fs, by_task


def _load_table(fs, paths: list[str]) -> pa.Table:
    """Read and cache the concatenated parquet table for one task's sample files."""
    key = "|".join(sorted(paths))
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


def list_sample_tasks(results_path: str | None) -> SampleTasksResponse:
    """Discover tasks with exported sample parquets under a run's results directory."""
    if not results_path:
        return SampleTasksResponse(available=False, error="run has no results_path", tasks=())
    try:
        _fs, by_task = _discover(results_path)
    except Exception as exc:
        logger.info("sample discovery failed for %s: %s", results_path, exc)
        return SampleTasksResponse(
            available=False,
            error=f"{type(exc).__name__}: {exc}"[:400],
            tasks=(),
        )
    tasks = tuple(SampleTask(task=task, files=len(paths)) for task, paths in sorted(by_task.items()))
    return SampleTasksResponse(available=True, error=None, tasks=tasks)


def _empty_samples(
    *,
    available: bool,
    error: str,
    task: str,
    offset: int,
    limit: int,
) -> SamplesResponse:
    return SamplesResponse(
        available=available,
        error=error,
        task=task,
        primary_metric=None,
        metric_columns=(),
        total=0,
        offset=offset,
        limit=limit,
        counts=SampleCounts(all=0, correct=0, incorrect=0),
        rows=(),
    )


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
        return _empty_samples(
            available=False,
            error="run has no results_path",
            task=task,
            offset=offset,
            limit=limit,
        )
    try:
        fs, by_task = _discover(results_path)
        paths = by_task.get(task)
        if not paths:
            return _empty_samples(
                available=True,
                error=f"no samples for task {task!r}",
                task=task,
                offset=offset,
                limit=limit,
            )
        table = _load_table(fs, paths)
    except Exception as exc:
        logger.info("sample fetch failed for %s/%s: %s", results_path, task, exc)
        return _empty_samples(
            available=False,
            error=f"{type(exc).__name__}: {exc}"[:400],
            task=task,
            offset=offset,
            limit=limit,
        )

    # Counts and the correctness filter need only the light ``correct`` and ``metrics`` columns, so
    # compute them straight from the arrow columns and validate just the page's rows. Validating every
    # row (and materializing every fat column, e.g. per-token logprobs) on each page request is what
    # made this reader scale with the whole run instead of the page.
    columns = set(table.column_names)
    correct_values = table.column("correct").to_pylist() if "correct" in columns else [None] * table.num_rows
    metric_maps = table.column("metrics").to_pylist() if "metrics" in columns else [None] * table.num_rows
    metric_columns = tuple(sorted({name for row in metric_maps if row for name in row}))
    picked = primary_metric(dict.fromkeys(metric_columns, 0.0))
    primary = picked[0] if picked is not None else None
    n_correct = sum(1 for value in correct_values if bool(value))
    counts = SampleCounts(
        all=table.num_rows,
        correct=n_correct,
        incorrect=table.num_rows - n_correct,
    )
    if correct == "correct":
        indices = [i for i, value in enumerate(correct_values) if bool(value)]
    elif correct == "incorrect":
        indices = [i for i, value in enumerate(correct_values) if not bool(value)]
    else:
        indices = list(range(table.num_rows))
    page_indices = indices[offset : offset + limit]
    page_rows = table.take(pa.array(page_indices, type=pa.int64())).to_pylist()
    page = tuple(EvalSample.model_validate(row) for row in page_rows)
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
    """True when ``uri`` sits under ``results_path`` with no upward traversal.

    The endpoint resolves only the ``trajectory_uri``/``exchange_uri`` a run wrote under its own
    results directory, so a URI on a different store, above the root, or reached through a ``..``
    segment is refused.
    """
    try:
        relative = StoragePath(uri).relative_to(StoragePath(results_path))
    except ValueError:
        return False
    return ".." not in relative.split("/")


def fetch_artifact(results_path: str | None, uri: str, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> ArtifactResponse:
    """Resolve one run-local artifact URI to decoded text, path-restricted and size-capped.

    Reads succeed only for URIs under ``results_path``; anything else -- a missing results directory,
    an out-of-tree URI, an object over ``max_bytes``, or an unreadable/unreachable object -- returns
    ``available=False`` with a reason instead of raising, so the caller degrades like the logs view.
    """
    if not results_path:
        return _unavailable_artifact(uri, "run has no results_path")
    if not _artifact_within_results(results_path, uri):
        logger.warning("artifact fetch rejected: %s is not under %s", uri, results_path)
        return _unavailable_artifact(uri, "artifact URI is outside the run results directory")

    cached = _artifact_cache.get(uri)
    if cached is not None:
        return cached

    # Resolve through StoragePath (the guarded url_to_fs/open_url factory) so an s3:// read inherits
    # finite socket timeouts and a cross-region read is budget-charged, the same as the path check above.
    path = StoragePath(uri)
    try:
        size = path.size()
        if size > max_bytes:
            return _unavailable_artifact(
                uri, f"artifact is {size} bytes; exceeds the {max_bytes}-byte cap", size=size, truncated=True
            )
        with path.open("rb") as handle:
            # Read one byte past the cap so a size the filesystem misreported is still caught.
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
    _artifact_cache.put(uri, response)
    return response
