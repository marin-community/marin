# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read lm-eval per-sample parquet exports for a run's results directory.

``marin.evaluation.sample_export`` writes one ``samples_<task>_<timestamp>.parquet`` per (sub)task
under a run's ``results_path``, each row a single evaluated question with its ``doc``/``target``/
``arguments``/``responses``/``filtered_responses`` (JSON strings) and per-sample metric columns
(``acc``, ``exact_match``, ...). This module discovers those files with fsspec (so ``gs://`` and
``s3://`` both work with the credentials the server already configures) and serves paginated,
correctness-filtered rows for the sample browser. Loaded tables are cached briefly so paging does not
re-read object storage on every request.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs
from metrics import primary_metric_column

logger = logging.getLogger(__name__)

SAMPLES_PREFIX = "samples_"
SAMPLES_SUFFIX = ".parquet"
# Non-metric columns written by sample_export; every other numeric column is a per-sample metric.
STRUCTURAL_COLUMNS = frozenset({"task", "doc_id", "doc", "target", "arguments", "responses", "filtered_responses"})
TABLE_CACHE_TTL = 120.0
CORRECT_THRESHOLD = 1.0


@dataclass
class _CachedTable:
    table: pa.Table
    expires_at: float


_cache: dict[str, _CachedTable] = {}
_cache_lock = threading.Lock()


def _sample_task(filename: str) -> str:
    """``samples_<task>_<timestamp>.parquet`` -> ``<task>`` (the timestamp has no underscore)."""
    stem = filename[len(SAMPLES_PREFIX) : -len(SAMPLES_SUFFIX)]
    return stem.rsplit("_", 1)[0]


def _discover(results_path: str):
    """Return ``(fs, {task: [parquet_path, ...]})`` for the sample files under ``results_path``."""
    fs, root = url_to_fs(results_path)
    by_task: dict[str, list[str]] = {}
    for path in fs.find(root):
        name = path.rsplit("/", 1)[-1]
        if name.startswith(SAMPLES_PREFIX) and name.endswith(SAMPLES_SUFFIX):
            by_task.setdefault(_sample_task(name), []).append(path)
    return fs, by_task


def _load_table(fs, paths: list[str]) -> pa.Table:
    """Read (and cache) the concatenated parquet table for one task's sample files."""
    key = "|".join(sorted(paths))
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None and cached.expires_at > now:
            return cached.table
    tables = []
    for path in sorted(paths):
        with fs.open(path, "rb") as handle:
            tables.append(pq.read_table(handle))
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
    with _cache_lock:
        _cache[key] = _CachedTable(table, now + TABLE_CACHE_TTL)
    return table


def list_sample_tasks(results_path: str | None) -> dict:
    """Discover which tasks have exported sample parquets under a run's results directory."""
    if not results_path:
        return {"available": False, "error": "run has no results_path", "tasks": []}
    try:
        _fs, by_task = _discover(results_path)
    except Exception as exc:
        logger.info("sample discovery failed for %s: %s", results_path, exc)
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:400], "tasks": []}
    tasks = [{"task": task, "files": len(paths)} for task, paths in sorted(by_task.items())]
    return {"available": True, "error": None, "tasks": tasks}


def _parse_json_cell(value) -> object:
    """Parse a JSON-string sample cell; leave a non-JSON string (e.g. a bare target) as-is."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _is_correct(row: dict, primary: str | None) -> bool:
    if primary is None:
        return False
    value = row.get(primary)
    return value is not None and float(value) >= CORRECT_THRESHOLD


def _sample_row(row: dict, metric_columns: list[str], primary: str | None) -> dict:
    return {
        "doc_id": row.get("doc_id"),
        "doc": _parse_json_cell(row.get("doc")),
        "target": _parse_json_cell(row.get("target")),
        "arguments": _parse_json_cell(row.get("arguments")),
        "responses": _parse_json_cell(row.get("responses")),
        "filtered_responses": _parse_json_cell(row.get("filtered_responses")),
        "metrics": {column: row.get(column) for column in metric_columns},
        "primary_value": row.get(primary) if primary else None,
        "correct": _is_correct(row, primary),
    }


def fetch_samples(results_path: str | None, task: str, *, offset: int, limit: int, correct: str) -> dict:
    """Paginated sample rows for one task, filtered by ``correct`` in ``{"correct","incorrect","all"}``."""
    if not results_path:
        return {"available": False, "error": "run has no results_path", "task": task, "total": 0, "rows": []}
    try:
        fs, by_task = _discover(results_path)
        paths = by_task.get(task)
        if not paths:
            return {"available": True, "error": f"no samples for task {task!r}", "task": task, "total": 0, "rows": []}
        table = _load_table(fs, paths)
    except Exception as exc:
        logger.info("sample fetch failed for %s/%s: %s", results_path, task, exc)
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:400], "task": task, "total": 0, "rows": []}

    metric_columns = [column for column in table.column_names if column not in STRUCTURAL_COLUMNS]
    primary = primary_metric_column(metric_columns)
    rows = table.to_pylist()
    if correct == "correct":
        rows = [row for row in rows if _is_correct(row, primary)]
    elif correct == "incorrect":
        rows = [row for row in rows if not _is_correct(row, primary)]
    total = len(rows)
    page = rows[offset : offset + limit]
    return {
        "available": True,
        "error": None,
        "task": task,
        "primary_metric": primary,
        "metric_columns": metric_columns,
        "total": total,
        "offset": offset,
        "limit": limit,
        "rows": [_sample_row(row, metric_columns, primary) for row in page],
    }
