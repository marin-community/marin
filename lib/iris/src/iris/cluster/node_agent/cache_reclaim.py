# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reclaim stale task-cache entries from a Kubernetes node."""

import logging
import os
import shutil
import threading
import uuid
from pathlib import Path

from rigging import telemetry
from rigging.timing import Duration, Timestamp

logger = logging.getLogger(__name__)

CACHE_RECLAIM_INTERVAL = Duration.from_minutes(5)
_CACHE_RECLAIM_PREFIX = ".iris-reclaim-"
_RECLAIM_FAILURES = telemetry.counter("iris_cache_reclaim_failures", unit="{failure}")


def _record_failure(failure_kind: str) -> None:
    _RECLAIM_FAILURES.add(1, attributes={"failure_kind": failure_kind})


def _entry_last_used(entry: Path) -> Timestamp:
    entry_stat = entry.lstat()
    latest = Timestamp.from_seconds(entry_stat.st_mtime)
    if entry.is_symlink() or not entry.is_dir():
        return Timestamp.from_seconds(max(entry_stat.st_mtime, entry_stat.st_atime))

    errors: list[OSError] = []
    for root, directories, files in os.walk(entry, followlinks=False, onerror=errors.append):
        # Walking a directory can update its atime, so only file atimes are a
        # useful signal that a task recently read the entry.
        for name in directories:
            try:
                latest = max(latest, Timestamp.from_seconds((Path(root) / name).lstat().st_mtime))
            except FileNotFoundError:
                continue
        for name in files:
            try:
                file_stat = (Path(root) / name).lstat()
                latest = max(latest, Timestamp.from_seconds(max(file_stat.st_mtime, file_stat.st_atime)))
            except FileNotFoundError:
                continue
    if errors:
        raise errors[0]
    return latest


def _remove_cache_entry(entry: Path) -> None:
    if entry.name.startswith(_CACHE_RECLAIM_PREFIX):
        tombstone = entry
    else:
        # A task can refill the original path without racing the slower
        # recursive deletion.
        tombstone = entry.with_name(f"{_CACHE_RECLAIM_PREFIX}{uuid.uuid4().hex}")
        entry.rename(tombstone)
    if tombstone.is_symlink() or not tombstone.is_dir():
        tombstone.unlink()
    else:
        shutil.rmtree(tombstone)


def reclaim_cache(
    cache_dir: Path,
    *,
    max_age: Duration,
    now: Timestamp | None = None,
) -> int:
    """Remove top-level cache entries older than ``max_age``."""
    if not cache_dir.exists():
        return 0

    cutoff = (now or Timestamp.now()).add_ms(-max_age.to_ms())
    reclaimed = 0
    for namespace in cache_dir.iterdir():
        if namespace.is_symlink() or not namespace.is_dir():
            continue
        for entry in namespace.iterdir():
            try:
                if not entry.name.startswith(_CACHE_RECLAIM_PREFIX) and _entry_last_used(entry) > cutoff:
                    continue
                _remove_cache_entry(entry)
                reclaimed += 1
            except FileNotFoundError:
                continue
            except OSError as error:
                logger.warning("could not reclaim cache entry %s: %s", entry, error)
                _record_failure("entry_removal")
    if reclaimed:
        logger.info("reclaimed %d stale cache entries from %s", reclaimed, cache_dir)
    return reclaimed


def run_cache_reclaimer(
    cache_dir: Path,
    max_age: Duration,
    stop: threading.Event,
) -> None:
    """Sweep the task cache periodically until the node agent stops."""
    while not stop.is_set():
        try:
            reclaim_cache(cache_dir, max_age=max_age)
        except OSError:
            logger.exception("cache reclamation failed for %s", cache_dir)
            _record_failure("cache_scan")
        stop.wait(CACHE_RECLAIM_INTERVAL.to_seconds())
