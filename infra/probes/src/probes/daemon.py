# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduler loop. Dispatches probes on worker threads, enforces deadlines
with a hard grace, emits scheduler-tick heartbeats.

Exit codes:
    0 — graceful shutdown via SIGTERM/SIGINT.
    1 — unrecoverable init error (SQLite quick_check failed, path invalid).
    2 — unrecoverable runtime error (SQLite write failed; supervisor restarts us).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from probes.probe import (
    ErrorClass,
    ProbeOutcome,
    ProbeResult,
    ProbeSample,
    ProbeSpec,
)
from probes.store.finelog import FinelogSampleStore, FinelogStoreConfig
from probes.store.sqlite import SqliteIntegrityError, SqliteSampleStore

logger = logging.getLogger(__name__)


_GRACE_SECONDS = 5.0
_HEARTBEAT_TICK_KIND = "heartbeat"
_HEARTBEAT_NAME = "scheduler"
DEFAULT_FINELOG_NAMESPACE = "marin.canary"
DEFAULT_META_NAMESPACE = "marin.canary.meta"


def run_canary(
    specs: list[ProbeSpec],
    *,
    sqlite_path: Path,
    finelog_endpoint: str | None = None,
    finelog_namespace: str = DEFAULT_FINELOG_NAMESPACE,
    finelog_meta_namespace: str = DEFAULT_META_NAMESPACE,
    heartbeat_seconds: int = 30,
    daemon_instance: str = "",
    once: bool = False,
) -> int:
    """Run the canary daemon.

    Args:
        specs: probes to run; must be non-empty with unique names.
        sqlite_path: absolute path to the SQLite store; parent dir must exist.
        finelog_endpoint: Finelog URL; None disables the secondary sink.
        once: dispatch each spec exactly once and exit; for tests + ops.

    Returns: process exit code.
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
                meta_namespace=finelog_meta_namespace,
            ),
            fallback_store=sqlite_store,
        )

    state = _SchedulerState(
        specs=specs,
        sqlite=sqlite_store,
        finelog=finelog_store,
        daemon_instance=daemon_instance or _default_daemon_instance(),
        heartbeat_seconds=heartbeat_seconds,
    )

    try:
        if once:
            _run_once(state)
            return 0
        return _run_forever(state)
    finally:
        sqlite_store.close()
        if finelog_store is not None:
            finelog_store.close()


class _SchedulerState:
    def __init__(
        self,
        *,
        specs: list[ProbeSpec],
        sqlite: SqliteSampleStore,
        finelog: FinelogSampleStore | None,
        daemon_instance: str,
        heartbeat_seconds: int,
    ):
        self.specs = specs
        self.sqlite = sqlite
        self.finelog = finelog
        self.daemon_instance = daemon_instance
        self.heartbeat_seconds = heartbeat_seconds
        self.abandoned_threads = 0
        self.loop_iteration = 0
        self.shutdown = threading.Event()


def _run_once(state: _SchedulerState) -> None:
    for spec in state.specs:
        sample = _execute(state, spec)
        _persist(state, sample)
    _persist(state, _heartbeat_sample(state, last_tick_ms=0), is_meta=True)


def _run_forever(state: _SchedulerState) -> int:
    _install_signal_handlers(state)

    next_run: dict[str, float] = {spec.name: time.monotonic() for spec in state.specs}
    next_heartbeat = time.monotonic()

    with ThreadPoolExecutor(max_workers=max(1, len(state.specs))) as executor:
        while not state.shutdown.is_set():
            state.loop_iteration += 1
            now = time.monotonic()
            tick_start = time.monotonic()

            ready = [s for s in state.specs if next_run[s.name] <= now]
            if ready:
                futures = {s.name: executor.submit(_execute, state, s) for s in ready}
                for spec in ready:
                    fut = futures[spec.name]
                    try:
                        sample = fut.result(timeout=spec.deadline_seconds + _GRACE_SECONDS)
                    except Exception as exc:
                        sample = _local_error_sample(state, spec, f"{type(exc).__name__}: {exc}")
                        state.abandoned_threads += 1
                    try:
                        _persist(state, sample)
                    except Exception:
                        logger.exception("fatal: failed to write probe sample to SQLite")
                        return 2
                    next_run[spec.name] = now + spec.cadence_seconds

            if time.monotonic() >= next_heartbeat:
                tick_ms = int((time.monotonic() - tick_start) * 1000)
                try:
                    _persist(state, _heartbeat_sample(state, last_tick_ms=tick_ms), is_meta=True)
                except Exception:
                    logger.exception("fatal: heartbeat write to SQLite failed")
                    return 2
                next_heartbeat = time.monotonic() + state.heartbeat_seconds

            now = time.monotonic()
            sleep_for = min(
                min(next_run.values()) - now if next_run else state.heartbeat_seconds,
                next_heartbeat - now,
                1.0,
            )
            if sleep_for > 0:
                state.shutdown.wait(timeout=sleep_for)

    return 0


def _execute(state: _SchedulerState, spec: ProbeSpec) -> ProbeSample:
    """Execute one probe, measure latency, build a ProbeSample. Never raises."""
    started_at = datetime.now(UTC)
    started_mono = time.monotonic()
    try:
        result = spec.probe.run(spec.deadline_seconds)
    except Exception as exc:
        logger.exception("probe %s leaked exception", spec.name)
        result = ProbeResult(
            outcome=ProbeOutcome.LOCAL_ERROR,
            error_class=ErrorClass.LOCAL_CONFIG_ERROR,
            error_detail=f"{type(exc).__name__}: {exc}",
        )
    latency_ms = int((time.monotonic() - started_mono) * 1000)

    # If a probe declares SUCCESS past its declared deadline, the daemon
    # promotes the outcome to TIMEOUT — mis-classifying late SUCCESS hides drift.
    outcome = result.outcome
    error_class = result.error_class
    error_detail = result.error_detail
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
        target_id=result.target_id,
        extras_json=json.dumps(result.extras or {}),
        daemon_instance=state.daemon_instance,
    )


def _local_error_sample(state: _SchedulerState, spec: ProbeSpec, detail: str) -> ProbeSample:
    """Worker thread itself failed to deliver a result within deadline + grace."""
    return ProbeSample(
        timestamp=datetime.now(UTC),
        probe_name=spec.name,
        probe_kind=spec.kind,
        location=spec.location,
        outcome=ProbeOutcome.TIMEOUT,
        latency_ms=int((spec.deadline_seconds + _GRACE_SECONDS) * 1000),
        error_class=ErrorClass.TIMEOUT,
        error_detail=detail,
        target_id=None,
        extras_json="{}",
        daemon_instance=state.daemon_instance,
    )


def _heartbeat_sample(state: _SchedulerState, *, last_tick_ms: int) -> ProbeSample:
    extras = {
        "disk_free_bytes": state.sqlite.disk_free_bytes(),
        "loop_iteration": state.loop_iteration,
        "abandoned_threads": state.abandoned_threads,
        "specs_count": len(state.specs),
    }
    return ProbeSample(
        timestamp=datetime.now(UTC),
        probe_name=_HEARTBEAT_NAME,
        probe_kind=_HEARTBEAT_TICK_KIND,
        location=None,
        outcome=ProbeOutcome.SUCCESS,
        latency_ms=last_tick_ms,
        error_class=None,
        error_detail=None,
        target_id=None,
        extras_json=json.dumps(extras),
        daemon_instance=state.daemon_instance,
    )


def _persist(state: _SchedulerState, sample: ProbeSample, *, is_meta: bool = False) -> None:
    """SQLite first (must succeed); Finelog secondary (best-effort)."""
    state.sqlite.write(sample)
    if state.finelog is not None:
        state.finelog.write(sample, is_meta=is_meta)


def _install_signal_handlers(state: _SchedulerState) -> None:
    def _handle(signum: int, _frame: object) -> None:
        logger.info("received signal %s, shutting down", signum)
        state.shutdown.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def _default_daemon_instance() -> str:
    return f"{socket.gethostname()}/{os.getpid()}/{int(time.time() * 1_000_000)}"
