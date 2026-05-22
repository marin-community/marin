# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Async per-probe loops. One coroutine per spec, run forever, persist
each sample, sleep cadence_seconds.

Probes are sync (their RPC clients are sync). We bridge to async via
``asyncio.to_thread`` and enforce a wall-clock deadline with
``asyncio.wait_for``. wait_for cancels the *coroutine*; the underlying
thread keeps running until the sync RPC returns. That's an honest
limitation — Python can't kill threads — but it gives us a TIMEOUT
sample on time even when the probe is wedged. A persistently wedged
probe leaks one ThreadPoolExecutor worker per cycle; the supervisor
is expected to recycle the process if that becomes pathological.

Exit codes:
    0 — graceful shutdown via SIGTERM/SIGINT.
    1 — unrecoverable init error (SQLite quick_check failed, path invalid).
    2 — unrecoverable runtime error (SQLite write failed; supervisor restarts us).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from probes.probe import (
    ErrorClass,
    ProbeOutcome,
    ProbeSample,
    ProbeSpec,
)
from probes.store.finelog import FinelogSampleStore, FinelogStoreConfig
from probes.store.sqlite import SqliteIntegrityError, SqliteSampleStore

logger = logging.getLogger(__name__)


_GRACE_SECONDS = 5.0
DEFAULT_FINELOG_NAMESPACE = "marin.canary"


def run_canary(
    specs: list[ProbeSpec],
    *,
    sqlite_path: Path,
    finelog_endpoint: str | None = None,
    finelog_namespace: str = DEFAULT_FINELOG_NAMESPACE,
    daemon_instance: str = "",
    once: bool = False,
) -> int:
    """Run the canary daemon.

    Blocks on the event loop until SIGTERM/SIGINT (or all probes finish, in
    --once mode). Returns process exit code.
    """
    if not specs:
        logger.error("no specs configured")
        return 1
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        logger.error("duplicate probe names: %s", sorted(names))
        return 1

    try:
        sqlite_store = SqliteSampleStore(sqlite_path)
    except SqliteIntegrityError as exc:
        logger.error("SQLite integrity check failed: %s", exc)
        return 1
    except (ValueError, OSError) as exc:
        logger.error("SQLite init failed: %s", exc)
        return 1

    finelog_store: FinelogSampleStore | None = None
    if finelog_endpoint:
        finelog_store = FinelogSampleStore(
            FinelogStoreConfig(
                endpoint=finelog_endpoint,
                probe_namespace=finelog_namespace,
            ),
            fallback_store=sqlite_store,
        )

    instance = daemon_instance or _default_daemon_instance()
    write_lock = threading.Lock()  # SQLite + LogClient called from to_thread workers

    try:
        return asyncio.run(_main(specs, sqlite_store, finelog_store, instance, write_lock, once))
    finally:
        sqlite_store.close()
        if finelog_store is not None:
            finelog_store.close()


async def _main(
    specs: list[ProbeSpec],
    sqlite: SqliteSampleStore,
    finelog: FinelogSampleStore | None,
    instance: str,
    write_lock: threading.Lock,
    once: bool,
) -> int:
    shutdown = asyncio.Event()

    if not once:
        _install_signal_handlers(shutdown)

    fatal: list[Exception] = []
    coros = [_probe_loop(spec, sqlite, finelog, instance, write_lock, shutdown, once, fatal) for spec in specs]
    await asyncio.gather(*coros)
    return 2 if fatal else 0


async def _probe_loop(
    spec: ProbeSpec,
    sqlite: SqliteSampleStore,
    finelog: FinelogSampleStore | None,
    instance: str,
    write_lock: threading.Lock,
    shutdown: asyncio.Event,
    once: bool,
    fatal: list[Exception],
) -> None:
    """Forever-loop one probe. Each iteration: run probe (with deadline+grace),
    persist sample, sleep cadence (or exit if shutdown). In --once mode, run
    exactly once and return."""
    while not shutdown.is_set():
        sample = await _execute(spec, instance)
        try:
            await asyncio.to_thread(_persist, sample, sqlite, finelog, write_lock)
        except Exception as exc:
            logger.exception("fatal: failed to write sample for %s", spec.name)
            fatal.append(exc)
            shutdown.set()
            return
        if once:
            return
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=spec.cadence_seconds)
        except asyncio.TimeoutError:
            pass  # cadence elapsed, next iteration


async def _execute(spec: ProbeSpec, instance: str) -> ProbeSample:
    """Run one probe with a hard deadline + grace. Always returns a sample."""
    started_at = datetime.now(UTC)
    started_mono = time.monotonic()
    outcome: ProbeOutcome
    error_class: ErrorClass | None
    error_detail: str | None
    target_id: str | None = None
    extras: dict[str, str | int | float] | None = None

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(spec.probe.run, spec.deadline_seconds),
            timeout=spec.deadline_seconds + _GRACE_SECONDS,
        )
    except asyncio.TimeoutError:
        outcome = ProbeOutcome.TIMEOUT
        error_class = ErrorClass.TIMEOUT
        error_detail = (
            f"daemon hard-deadline reached at {int((spec.deadline_seconds + _GRACE_SECONDS) * 1000)} ms; "
            "the worker thread may still be running"
        )
    except Exception as exc:
        logger.exception("probe %s leaked exception", spec.name)
        outcome = ProbeOutcome.LOCAL_ERROR
        error_class = ErrorClass.LOCAL_CONFIG_ERROR
        error_detail = f"{type(exc).__name__}: {exc}"
    else:
        outcome = result.outcome
        error_class = result.error_class
        error_detail = result.error_detail
        target_id = result.target_id
        extras = result.extras

    latency_ms = int((time.monotonic() - started_mono) * 1000)

    # A probe that returns SUCCESS past its declared deadline is suspect;
    # mis-classifying late SUCCESS hides drift.
    if outcome is ProbeOutcome.SUCCESS and latency_ms > spec.deadline_seconds * 1000:
        outcome = ProbeOutcome.TIMEOUT
        error_class = ErrorClass.TIMEOUT
        error_detail = f"probe returned after {latency_ms} ms (deadline {int(spec.deadline_seconds * 1000)} ms)"

    return ProbeSample(
        timestamp=started_at,
        probe_name=spec.name,
        probe_kind=spec.kind,
        location=spec.location,
        outcome=outcome,
        latency_ms=latency_ms,
        error_class=error_class,
        error_detail=error_detail,
        target_id=target_id,
        extras_json=json.dumps(extras or {}),
        daemon_instance=instance,
    )


def _persist(
    sample: ProbeSample,
    sqlite: SqliteSampleStore,
    finelog: FinelogSampleStore | None,
    write_lock: threading.Lock,
) -> None:
    """SQLite first (must succeed); Finelog secondary (best-effort).

    The lock serializes writes from concurrent ``to_thread`` workers; SQLite
    is opened with ``check_same_thread=False`` but only one writer at a time
    is supported under WAL.
    """
    with write_lock:
        sqlite.write(sample)
        if finelog is not None:
            finelog.write(sample)


def _install_signal_handlers(shutdown: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def _handle(signum: int) -> None:
        logger.info("received signal %s, shutting down", signum)
        shutdown.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle, sig)
        except NotImplementedError:
            # add_signal_handler is unix-only; fall back to signal.signal for portability.
            signal.signal(sig, lambda s, _f: shutdown.set())


def _default_daemon_instance() -> str:
    return f"{socket.gethostname()}/{os.getpid()}/{int(time.time() * 1_000_000)}"
