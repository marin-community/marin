# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Distributed-locked caching of model snapshots to a shared prefix.

Downloading a large model repo from HuggingFace on every job is slow and wastes
bandwidth. :func:`cache_to_prefix` mirrors a snapshot once to a shared cache
prefix (typically a region-local TTL bucket from
:func:`rigging.filesystem.marin_temp_bucket`) under a distributed lock, so
concurrent workers do not all hammer HuggingFace at the same time: the first
worker populates the cache while the rest **block** until it is complete, then
read the snapshot from the nearby cache.

The populate step streams **directly into the cache filesystem** — there is no
full local copy of the snapshot. :func:`cache_hf_model` is the easy path for the
common case (mirror an HF repo by id), downloading and uploading one file at a
time so a multi-hundred-GB repo never has to fit on local disk. :func:`cache_to_prefix`
takes an arbitrary populate callback for callers that need a custom source.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Generator

import fsspec
from huggingface_hub import hf_hub_download, list_repo_files
from rigging.filesystem.distributed_lock import HEARTBEAT_INTERVAL, DistributedLease, LeaseLostError, create_lock
from rigging.filesystem import marin_temp_bucket, url_to_fs

logger = logging.getLogger(__name__)

DEFAULT_COMPLETE_MARKER = ".cache_complete"
"""Marker written last after a full upload; its presence is the cache-hit signal."""

_LOCK_SUFFIX = ".lock"
# How often losers re-check the completion marker while the winner populates.
_DEFAULT_POLL_INTERVAL = 10.0

# A populate callback streams the snapshot into ``cache_path`` on the given
# filesystem. It must write every file but NOT the completion marker (the cache
# writes that last so a crashed populate never reads as a hit).
Populate = Callable[[fsspec.AbstractFileSystem, str], None]


def cache_to_prefix(
    cache_path: str,
    populate: Populate,
    *,
    complete_marker: str = DEFAULT_COMPLETE_MARKER,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> str:
    """Populate *cache_path* once under a distributed lock and return it.

    Fast path: if the completion marker already exists, return ``cache_path``
    immediately. Otherwise acquire a distributed lock keyed on ``cache_path``:

    - The **winner** runs ``populate`` (which streams files straight into the
      cache filesystem), writes the marker last (so a crashed populate never
      reads as a hit), and returns ``cache_path``.
    - **Losers** block, polling for the marker every ``poll_interval`` seconds,
      then return ``cache_path`` once the winner finishes. If the holder dies
      without completing (its lease goes stale), a loser takes over and becomes
      the new winner.

    Args:
        cache_path: Destination prefix for the mirrored snapshot (any fsspec
            path, e.g. ``gs://…`` or a local directory).
        populate: Callback ``(fs, cache_path) -> None`` that streams the snapshot
            into ``cache_path`` on filesystem ``fs``. Use :func:`cache_hf_model`
            for the common HF-repo case instead of writing one by hand.
        complete_marker: Filename (relative to ``cache_path``) written last to
            mark the cache complete.
        poll_interval: Seconds a blocked worker waits between marker re-checks.

    Returns:
        ``cache_path`` — a path the caller can load the snapshot from.
    """
    cache_path = cache_path.rstrip("/")
    fs, _ = url_to_fs(cache_path)
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
            _populate(fs, cache_path, marker, populate)
        return cache_path
    finally:
        lock.release()


def cache_hf_model(
    cache_path: str,
    model_id: str,
    *,
    revision: str | None = None,
    complete_marker: str = DEFAULT_COMPLETE_MARKER,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> str:
    """Mirror HuggingFace repo *model_id* to *cache_path* once under a distributed lock.

    The easy path over :func:`cache_to_prefix`: streams the repo into the cache
    one file at a time (see :func:`_stream_hf_snapshot`), so the snapshot never
    has to fit on local disk. Returns ``cache_path``.

    Args:
        cache_path: Destination prefix for the mirrored snapshot.
        model_id: HuggingFace repo id, e.g. ``Qwen/Qwen3-0.6B``.
        revision: Optional git revision (branch, tag, or commit) to mirror.
        complete_marker: Filename written last to mark the cache complete.
        poll_interval: Seconds a blocked worker waits between marker re-checks.
    """
    return cache_to_prefix(
        cache_path,
        lambda fs, dest: _stream_hf_snapshot(fs, dest, model_id, revision),
        complete_marker=complete_marker,
        poll_interval=poll_interval,
    )


def resolve_cached_model_path(
    model: str,
    *,
    cache_ttl_days: int,
    cache_prefix: str,
    complete_marker: str = DEFAULT_COMPLETE_MARKER,
) -> str:
    """Resolve *model* to a path to load it from, mirroring a HuggingFace repo to a TTL cache.

    A bare HuggingFace repo id (optionally ``org/model@revision``) is mirrored once to a
    region-local TTL bucket under a distributed lock and the cache path is returned, so later
    loads of the same model read the snapshot from nearby storage instead of re-downloading
    from HuggingFace. Object-store and local paths already name a loadable snapshot and are
    returned unchanged, as is *model* when ``cache_ttl_days`` is non-positive (mirroring
    disabled).

    Args:
        model: HuggingFace repo id (``org/model`` or ``org/model@revision``), or an
            object-store / local path holding a snapshot.
        cache_ttl_days: TTL in days for the region-local mirror; ``<= 0`` disables mirroring.
        cache_prefix: Sub-path under the TTL bucket the mirror is written to.
        complete_marker: Filename written last to mark the cache complete.

    Returns:
        A path the caller can load *model* from.
    """
    # Deferred so importing levanter (which eagerly imports this module) does not pull
    # transformers via hf_checkpoints onto CPU-only workers (see levanter/__init__.py).
    from levanter.compat.hf_checkpoints import _is_hf_model_id  # noqa: PLC0415  # optional dep: transformers

    repo, _, revision = model.partition("@")
    if cache_ttl_days <= 0 or not _is_hf_model_id(repo):
        return model

    cache_path = marin_temp_bucket(cache_ttl_days, f"{cache_prefix}/{_cache_slug(model)}")
    return cache_hf_model(cache_path, repo, revision=revision or None, complete_marker=complete_marker)


def _cache_slug(model: str) -> str:
    """Collision-resistant, filesystem-safe cache slug for a model ref.

    The sanitized prefix keeps the cache dir readable when browsing the bucket; the appended
    digest of the full ref keeps distinct refs in distinct dirs even when sanitizing alone
    would collide them (e.g. ``org/model_a`` and ``org/model@a`` both sanitize to
    ``org_model_a``).
    """
    ref = model.strip("/")
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", ref)
    digest = hashlib.sha256(ref.encode()).hexdigest()[:16]
    return f"{readable}-{digest}"


def _stream_hf_snapshot(fs: fsspec.AbstractFileSystem, dest: str, model_id: str, revision: str | None) -> None:
    """Copy every file of HF repo *model_id* into *dest*, one file at a time.

    Each file is downloaded to a scratch dir, uploaded to ``dest``, then deleted
    before the next download, so peak local disk is one file rather than the full
    repo.
    """
    filenames = list_repo_files(model_id, revision=revision)
    logger.info("streaming %d files from HF repo %s into %s", len(filenames), model_id, dest)
    with tempfile.TemporaryDirectory(prefix="hf_stream_") as scratch:
        for filename in filenames:
            local_path = hf_hub_download(model_id, filename, revision=revision, local_dir=scratch)
            remote_path = f"{dest}/{filename}"
            # Local/posix-backed fsspec filesystems don't auto-create parents; object
            # stores treat this as a no-op since they have no real directories.
            fs.makedirs(remote_path.rsplit("/", 1)[0], exist_ok=True)
            fs.put_file(local_path, remote_path)
            os.remove(local_path)


def _populate(fs: fsspec.AbstractFileSystem, cache_path: str, marker: str, populate: Populate) -> None:
    """Stream the snapshot into *cache_path* via *populate*, then write *marker*."""
    logger.info("model cache miss; populating %s", cache_path)
    # Object stores have no real directories, but local/posix-backed fsspec
    # filesystems need the prefix to exist before files are written into it.
    fs.makedirs(cache_path, exist_ok=True)
    populate(fs, cache_path)
    # Marker last: its presence is the cache-hit signal, so a crashed populate won't read as complete.
    with fs.open(marker, "w") as handle:
        handle.write("ok")


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
