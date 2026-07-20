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
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs
from marin.evaluation.samples import SAMPLES_PREFIX, SAMPLES_SUFFIX, EvalSample, primary_metric
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

TABLE_CACHE_TTL = 120.0


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


@dataclass
class _CachedTable:
    table: pa.Table
    expires_at: float


_cache: dict[str, _CachedTable] = {}
_cache_lock = threading.Lock()


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
    now = time.monotonic()
    with _cache_lock:
        for expired_key in [cache_key for cache_key, value in _cache.items() if value.expires_at <= now]:
            _cache.pop(expired_key)
        cached = _cache.get(key)
        if cached is not None:
            return cached.table
    tables = []
    for path in sorted(paths):
        with fs.open(path, "rb") as handle:
            tables.append(pq.read_table(handle))
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
    with _cache_lock:
        _cache[key] = _CachedTable(table, now + TABLE_CACHE_TTL)
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


def _correct(sample: EvalSample) -> bool:
    return bool(sample.correct)


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

    all_samples = tuple(EvalSample.model_validate(row) for row in table.to_pylist())
    metric_columns = tuple(sorted({name for sample in all_samples for name in sample.metrics}))
    picked = primary_metric(dict.fromkeys(metric_columns, 0.0))
    primary = picked[0] if picked is not None else None
    n_correct = sum(1 for sample in all_samples if _correct(sample))
    counts = SampleCounts(
        all=len(all_samples),
        correct=n_correct,
        incorrect=len(all_samples) - n_correct,
    )
    if correct == "correct":
        filtered = tuple(sample for sample in all_samples if _correct(sample))
    elif correct == "incorrect":
        filtered = tuple(sample for sample in all_samples if not _correct(sample))
    else:
        filtered = all_samples
    page = filtered[offset : offset + limit]
    return SamplesResponse(
        available=True,
        error=None,
        task=task,
        primary_metric=primary,
        metric_columns=metric_columns,
        total=len(filtered),
        offset=offset,
        limit=limit,
        counts=counts,
        rows=page,
    )
