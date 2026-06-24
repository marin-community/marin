# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Distributed-locked caching of model snapshots to a shared prefix.

Downloading a large model repo from HuggingFace on every job is slow and wastes
bandwidth. :func:`cache_to_prefix` mirrors a snapshot once to a shared cache
prefix (typically a region-local TTL bucket from
:func:`rigging.filesystem.marin_temp_bucket`) under a distributed lock, so
concurrent workers do not all hammer HuggingFace at the same time: the first
worker downloads and uploads while the rest **block** until the cache is
populated, then read the snapshot from the nearby cache.

The download is injected as a callback so this module stays free of any
``huggingface_hub`` dependency; callers bind ``snapshot_download`` (or any other
populator) themselves.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import threading
import time
from collections.abc import Callable, Generator

import fsspec
from rigging.distributed_lock import HEARTBEAT_INTERVAL, DistributedLease, LeaseLostError, create_lock

logger = logging.getLogger(__name__)

DEFAULT_COMPLETE_MARKER = ".cache_complete"
"""Marker written last after a full upload; its presence is the cache-hit signal."""

_LOCK_SUFFIX = ".lock"
# How often losers re-check the completion marker while the winner downloads.
_DEFAULT_POLL_INTERVAL = 10.0


def cache_to_prefix(
    cache_path: str,
    download: Callable[[str], None],
    *,
    complete_marker: str = DEFAULT_COMPLETE_MARKER,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> str:
    """Populate *cache_path* once under a distributed lock and return a loadable path.

    Fast path: if the completion marker already exists, return ``cache_path``
    immediately. Otherwise acquire a distributed lock keyed on ``cache_path``:

    - The **winner** downloads into a fresh local temp dir via ``download``,
      uploads the result to ``cache_path``, writes the marker last (so a crashed
      upload never reads as a hit), and returns the **local temp dir** — a fast
      local read that avoids re-reading what it just uploaded.
    - **Losers** block, polling for the marker every ``poll_interval`` seconds,
      then return ``cache_path`` once the winner finishes. If the holder dies
      without completing (its lease goes stale), a loser takes over and becomes
      the new winner.

    Args:
        cache_path: Destination prefix for the mirrored snapshot (any fsspec
            path, e.g. ``gs://…`` or a local directory).
        download: Callback that populates the local directory passed to it with
            the files to cache. Typically binds ``snapshot_download``.
        complete_marker: Filename (relative to ``cache_path``) written last to
            mark the cache complete.
        poll_interval: Seconds a blocked worker waits between marker re-checks.

    Returns:
        A path that the caller can load the snapshot from: the local temp dir for
        the worker that populated the cache, or ``cache_path`` otherwise.
    """
    cache_path = cache_path.rstrip("/")
    fs, _ = fsspec.core.url_to_fs(cache_path)
    marker = f"{cache_path}/{complete_marker}"

    if fs.exists(marker):
        logger.info("model cache hit: %s", cache_path)
        return cache_path

    lock = create_lock(f"{cache_path}{_LOCK_SUFFIX}")
    while not lock.try_acquire():
        # Another worker is populating the cache; block then re-check the marker
        # rather than piling another download onto HuggingFace.
        if fs.exists(marker):
            logger.info("model cache populated by another worker: %s", cache_path)
            return cache_path
        time.sleep(poll_interval)

    try:
        # A prior holder may have finished while we were acquiring the lock.
        if fs.exists(marker):
            logger.info("model cache populated while acquiring lock: %s", cache_path)
            return cache_path
        with _heartbeat(lock):
            return _populate(fs, cache_path, marker, download)
    finally:
        lock.release()


def _populate(fs: fsspec.AbstractFileSystem, cache_path: str, marker: str, download: Callable[[str], None]) -> str:
    """Download to a local temp dir, mirror it to *cache_path*, then write *marker*."""
    logger.info("model cache miss; downloading to populate %s", cache_path)
    local_dir = tempfile.mkdtemp(prefix="model_cache_")
    download(local_dir)
    fs.put(f"{local_dir.rstrip('/')}/", f"{cache_path}/", recursive=True)
    # Marker last: its presence is the cache-hit signal, so a crashed upload won't read as complete.
    with fs.open(marker, "w") as handle:
        handle.write("ok")
    return local_dir


@contextlib.contextmanager
def _heartbeat(lock: DistributedLease) -> Generator[None, None, None]:
    """Refresh *lock*'s lease on a daemon thread for the duration of the block.

    A large download can take far longer than ``HEARTBEAT_TIMEOUT``; without
    refreshing, blocked workers would treat the lease as stale and start their
    own downloads. ``LeaseLostError`` is logged but not raised: a duplicate
    download is wasteful but not incorrect (the marker-last write stays atomic).
    """
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(HEARTBEAT_INTERVAL):
            try:
                lock.refresh()
            except LeaseLostError:
                logger.error("Lease lost for %s while populating cache", lock.lock_path, exc_info=True)
                return

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)
