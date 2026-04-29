# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-time migration from finelog's flat parquet layout to the per-namespace layout.

The old layout placed every segment directly under ``data_dir`` as either
``tmp_{seq}.parquet`` or ``logs_{seq}.parquet``. The stats service introduces
per-namespace subdirectories. The single existing namespace, ``log``, gets its
files relocated into ``{data_dir}/log/`` on first startup.

The migration runs synchronously inside :func:`finelog.server.main.run_log_server`
*before* the log store is instantiated and *before* uvicorn binds — callers
either fail to connect or block on the listening socket until startup is done.

Sentinel state machine (file: ``{data_dir}/.layout-migration``)
---------------------------------------------------------------

The sentinel is a single-line JSON object::

    {"version": 1, "state": "<state>", "started_at": <ms>, "finished_at": <ms|null>}

States:
    ``in-progress`` — a migration is running or crashed mid-walk.
    ``done`` — the directory is in the per-namespace layout. Steady state.

A missing sentinel means "unknown — inspect the directory" (cold start path).

The walk is fully idempotent: each segment is moved with :func:`os.rename`,
which is atomic within one POSIX filesystem. A crash leaves each file either
fully at the source or fully at the destination, so re-running the walk
converges. The next run must be identical, so we never journal individual
moves — the (src, dst) pair *is* the journal.

Pre-flight refuses to migrate when the destination exists but is not a
directory, or when ``data_dir`` and ``data_dir/log`` resolve to different
filesystems (cross-mount rename is not atomic on POSIX).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SENTINEL_FILENAME = ".layout-migration"
LOG_NAMESPACE_DIR = "log"
SENTINEL_VERSION = 1

_TMP_GLOB = "tmp_*.parquet"
_LOG_GLOB = "logs_*.parquet"

_STATE_IN_PROGRESS = "in-progress"
_STATE_DONE = "done"

# Frequency of the per-batch progress log line during a long migration walk.
_PROGRESS_LOG_INTERVAL = 500


def migrate_to_namespaced_layout(data_dir: Path) -> None:
    """Migrate finelog's flat parquet layout under ``data_dir`` into per-namespace
    subdirectories. Idempotent and crash-safe.

    Behavior:
        - Fast path: if the sentinel exists with state ``done``, return immediately
          without scanning the directory.
        - Cold start: if the sentinel is missing and no flat files / no ``log/``
          subdir exist, this is a fresh install — create ``log/`` and write
          the sentinel.
        - Otherwise run the idempotent walk that moves ``tmp_*.parquet`` and
          ``logs_*.parquet`` from ``{data_dir}/`` into ``{data_dir}/log/``.

    Raises:
        RuntimeError: ``{data_dir}/log`` exists but is not a directory, or
            ``{data_dir}/log`` lives on a different filesystem than ``{data_dir}``,
            or a duplicate file was found at the destination with a different
            size during a resume.
    """
    data_dir = Path(data_dir)
    sentinel = data_dir / SENTINEL_FILENAME

    state = _read_sentinel_state(sentinel)
    if state == _STATE_DONE:
        return

    log_dir = data_dir / LOG_NAMESPACE_DIR
    if log_dir.exists() and not log_dir.is_dir():
        raise RuntimeError(f"{log_dir} exists but is not a directory; refusing to migrate")

    flat = sorted(data_dir.glob(_TMP_GLOB)) + sorted(data_dir.glob(_LOG_GLOB))

    if not flat and not log_dir.exists():
        # Fresh install: nothing to move, just create the namespace dir and
        # mark migration done.
        log_dir.mkdir(parents=True, exist_ok=True)
        _write_sentinel(sentinel, state=_STATE_DONE, started_at_ms=_now_ms(), finished_at_ms=_now_ms())
        return

    if not flat and log_dir.exists():
        # Existing per-namespace layout (or a partially-migrated dir whose
        # crash left zero flat files). Just stamp the sentinel.
        _write_sentinel(sentinel, state=_STATE_DONE, started_at_ms=_now_ms(), finished_at_ms=_now_ms())
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    _assert_same_filesystem(data_dir, log_dir)

    started_at_ms = _now_ms()
    _write_sentinel(sentinel, state=_STATE_IN_PROGRESS, started_at_ms=started_at_ms, finished_at_ms=None)

    logger.info("layout migration: %d flat segments to move into %s", len(flat), log_dir)
    moved = 0
    skipped = 0
    for i, src in enumerate(flat, start=1):
        dst = log_dir / src.name
        if dst.exists():
            # Prior crashed run already moved this one. Source is the
            # duplicate; sizes must match or we abort loudly.
            src_size = src.stat().st_size
            dst_size = dst.stat().st_size
            if src_size != dst_size:
                raise RuntimeError(
                    f"layout migration: size mismatch on resume for {src.name}: " f"src={src_size} dst={dst_size}"
                )
            src.unlink()
            skipped += 1
        else:
            os.rename(src, dst)
            moved += 1
        if i % _PROGRESS_LOG_INTERVAL == 0:
            logger.info("layout migration: %d/%d processed", i, len(flat))

    _write_sentinel(sentinel, state=_STATE_DONE, started_at_ms=started_at_ms, finished_at_ms=_now_ms())
    logger.info("layout migration: complete (moved=%d skipped=%d)", moved, skipped)


def _read_sentinel_state(sentinel: Path) -> str | None:
    """Return the ``state`` field from the sentinel, or ``None`` if missing.

    A malformed sentinel is treated as ``None`` (re-run the migration). We do
    not silently overwrite — the caller's walk is idempotent, so a re-run is
    safe and will rewrite the sentinel on completion.
    """
    if not sentinel.exists():
        return None
    raw = sentinel.read_text()
    payload = json.loads(raw)
    state = payload.get("state")
    if state not in (_STATE_IN_PROGRESS, _STATE_DONE):
        return None
    return state


def _write_sentinel(sentinel: Path, *, state: str, started_at_ms: int, finished_at_ms: int | None) -> None:
    """Atomically write the sentinel via ``os.replace`` of a sibling tmp file."""
    payload = {
        "version": SENTINEL_VERSION,
        "state": state,
        "started_at": started_at_ms,
        "finished_at": finished_at_ms,
    }
    tmp = sentinel.with_suffix(sentinel.suffix + ".tmp")
    tmp.write_text(json.dumps(payload) + "\n")
    os.replace(tmp, sentinel)


def _assert_same_filesystem(parent: Path, child: Path) -> None:
    parent_dev = os.stat(parent).st_dev
    child_dev = os.stat(child).st_dev
    if parent_dev != child_dev:
        raise RuntimeError(
            f"layout migration: {parent} (st_dev={parent_dev}) and {child} (st_dev={child_dev}) "
            f"are on different filesystems; cross-mount rename is not atomic"
        )


def _now_ms() -> int:
    return int(time.time() * 1000)
