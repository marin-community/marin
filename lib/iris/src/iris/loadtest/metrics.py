# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Time-series metric collection for load-test scenarios.

The harness exposes the moving pieces (autoscaler, DB, fake GCP service);
this module wraps them to produce per-second samples:

- ``active_scale_up_threads`` — live threads in ``Autoscaler._threads``.
- ``scale_up_pending`` — sum of ``ScalingGroup._pending_scale_ups``.
- ``create_attempts`` / ``create_failures`` — from ``LoadtestGcpService``.
- ``writer_lock_hold_ms`` — samples collected by wrapping the ``ControllerDB``
  writer lock with an instrumented context manager, reservoir-bounded.
- ``dashboard_query_ms`` — latency of a representative read issued by a
  dedicated probe thread at ~2 Hz.
- ``rss_bytes`` — ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (delta vs
  scenario start).

The writer-lock instrumentation monkey-patches ``db._lock`` on start and
restores it on stop — no changes to production code, no subclassing.
"""

from __future__ import annotations

import json
import logging
import resource
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


RESERVOIR_SIZE = 1000


def _rss_bytes() -> int:
    """Return maxrss in bytes (Linux reports KB, macOS reports bytes)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(ru)
    return int(ru) * 1024


@dataclass
class _Reservoir:
    """Bounded deque of recent float samples plus a running total count."""

    capacity: int = RESERVOIR_SIZE
    _samples: deque = field(default_factory=lambda: deque(maxlen=RESERVOIR_SIZE))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    total: int = 0

    def add(self, value: float) -> None:
        with self._lock:
            self._samples.append(value)
            self.total += 1

    def snapshot(self) -> list[float]:
        with self._lock:
            return list(self._samples)


def _percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0, "max": 0.0}
    ordered = sorted(samples)
    n = len(ordered)

    def q(p: float) -> float:
        idx = min(n - 1, max(0, round(p * (n - 1))))
        return ordered[idx]

    return {
        "p50": q(0.5),
        "p95": q(0.95),
        "p99": q(0.99),
        "count": n,
        "max": ordered[-1],
    }


class _InstrumentedLock:
    """Wrapper around an RLock that records hold time (ms) on release.

    ``threading.RLock`` is a factory, not a class, so we can't subclass it
    directly. Instead we wrap the original lock and forward ``acquire`` /
    ``release`` / the context-manager protocol. Hold time is measured from
    the time the underlying ``acquire`` returns to the corresponding
    ``release``; re-entrant acquires don't double-count (only the outermost
    hold records a sample on final release).
    """

    def __init__(self, inner, reservoir: _Reservoir) -> None:
        self._inner = inner
        self._reservoir = reservoir
        self._tl = threading.local()

    def _depth(self) -> int:
        return getattr(self._tl, "depth", 0)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        # RLock.acquire accepts (blocking, timeout) but positional timeout
        # requires blocking=True. Forward precisely what we got.
        if timeout == -1:
            got = self._inner.acquire(blocking)
        else:
            got = self._inner.acquire(blocking, timeout)
        if not got:
            return False
        depth = self._depth()
        if depth == 0:
            self._tl.start = time.monotonic()
        self._tl.depth = depth + 1
        return True

    def release(self) -> None:
        depth = self._depth()
        if depth == 1:
            elapsed_ms = (time.monotonic() - getattr(self._tl, "start", time.monotonic())) * 1000.0
            self._reservoir.add(elapsed_ms)
        self._tl.depth = max(0, depth - 1)
        self._inner.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


@dataclass
class _Sample:
    t: float
    active_scale_up_threads: int
    scale_up_pending: int
    create_attempts: int
    create_failures: int
    rss_bytes: int


class ScenarioMetrics:
    """Collects time-series metrics for a single scenario run.

    Lifecycle:
        m = ScenarioMetrics(harness)
        m.start()
        ... run stimuli ...
        m.stop()
        m.dump_json(path)
    """

    def __init__(self, harness, *, sample_interval_s: float = 1.0, probe_interval_s: float = 0.5) -> None:
        self._harness = harness
        self._sample_interval_s = sample_interval_s
        self._probe_interval_s = probe_interval_s
        self._stop = threading.Event()
        self._samples: list[_Sample] = []
        self._samples_lock = threading.Lock()
        self._lock_hold = _Reservoir()
        self._query_latency = _Reservoir()
        # Per-method RPC wall-clock reservoirs. Populated when start() wraps
        # the harness's WorkerProvider. Key is the method name
        # (e.g. "ping_workers", "sync", "start_tasks", "stop_tasks",
        # "poll_workers"); value is elapsed ms for one call to that method.
        self._rpc: dict[str, _Reservoir] = {}
        self._started_at: float | None = None
        self._rss_start: int = 0
        self._sampler_thread: threading.Thread | None = None
        self._probe_thread: threading.Thread | None = None
        self._original_db_lock: object | None = None
        self._original_read_pool_entries: list = []
        # (provider, method_name, original_callable) tuples restored on stop().
        self._rpc_patches: list[tuple[Any, str, Any]] = []

    def start(self) -> None:
        # Swap the DB writer lock with an instrumented one. ControllerDB uses
        # `self._lock` everywhere (including QuerySnapshot passed a reference
        # in `snapshot()`), so we patch it in place and restore on stop. The
        # instrumented wrapper is ABI-compatible: `with self._lock: ...` and
        # explicit acquire/release both work.
        db = self._harness.db
        self._original_db_lock = db._lock
        db._lock = _InstrumentedLock(self._original_db_lock, self._lock_hold)

        self._started_at = time.monotonic()
        self._rss_start = _rss_bytes()

        self._wrap_rpc_methods()

        self._sampler_thread = threading.Thread(target=self._run_sampler, name="loadtest-sampler", daemon=True)
        self._probe_thread = threading.Thread(target=self._run_probe, name="loadtest-probe", daemon=True)
        self._sampler_thread.start()
        self._probe_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for t in (self._sampler_thread, self._probe_thread):
            if t is not None:
                t.join(timeout=5.0)
        # Restore the original lock so the harness's own shutdown path runs
        # against the production lock (avoids skewing teardown-time metrics).
        if self._original_db_lock is not None:
            self._harness.db._lock = self._original_db_lock
            self._original_db_lock = None
        # Restore original RPC methods.
        for obj, name, original in self._rpc_patches:
            setattr(obj, name, original)
        self._rpc_patches.clear()

    def _wrap_rpc_methods(self) -> None:
        """Wrap WorkerProvider entrypoints with wall-clock timing.

        Batch-level granularity: one sample per ``sync`` / ``ping_workers``
        / ``start_tasks`` / ``stop_tasks`` / ``poll_workers`` invocation. The
        value is the method's total wall time in ms, which is the end-to-end
        cost the calling loop actually pays per tick. Per-RPC-per-worker
        timing would need to hook the inner coroutines; batch-level is
        enough to distinguish slow legacy heartbeat sync (10-100 s) from
        split-heartbeat (<1 s).
        """
        provider = getattr(self._harness, "_task_provider", None)
        if provider is None:
            return
        for method_name in ("sync", "ping_workers", "start_tasks", "stop_tasks", "poll_workers"):
            original = getattr(provider, method_name, None)
            if original is None:
                continue
            reservoir = self._rpc.setdefault(method_name, _Reservoir())

            def make_wrapper(fn, name, res):
                def wrapper(*args, **kwargs):
                    t0 = time.monotonic()
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        res.add((time.monotonic() - t0) * 1000.0)

                wrapper.__name__ = name
                return wrapper

            setattr(provider, method_name, make_wrapper(original, method_name, reservoir))
            self._rpc_patches.append((provider, method_name, original))

    def _run_sampler(self) -> None:
        while not self._stop.is_set():
            sample = self._take_sample()
            with self._samples_lock:
                self._samples.append(sample)
            if self._stop.wait(self._sample_interval_s):
                return

    def _run_probe(self) -> None:
        # Issue the representative dashboard query. Use `read_snapshot` so we
        # exercise the same read path the dashboard uses; this uses a pooled
        # read-only connection and does NOT take the writer lock, which is
        # exactly what we want to measure under writer-lock contention.
        while not self._stop.is_set():
            try:
                t0 = time.monotonic()
                with self._harness.db.read_snapshot() as q:
                    q.execute_sql("SELECT state, count(*) FROM tasks GROUP BY state").fetchall()
                self._query_latency.add((time.monotonic() - t0) * 1000.0)
            except Exception:
                logger.exception("probe query failed")
            if self._stop.wait(self._probe_interval_s):
                return

    def _take_sample(self) -> _Sample:
        autoscaler = self._harness.autoscaler
        with autoscaler._threads._lock:
            active_threads = len(autoscaler._threads._threads)
        pending = 0
        for g in autoscaler._groups.values():
            # Read without acquiring ScalingGroup's lock — it's a single
            # int assignment in Python and we're sampling, not enforcing a
            # consistency invariant across groups.
            pending += getattr(g, "_pending_scale_ups", 0)
        gcp = self._harness.gcp_service
        return _Sample(
            t=time.monotonic() - (self._started_at or 0.0),
            active_scale_up_threads=active_threads,
            scale_up_pending=pending,
            create_attempts=gcp.counts_create_attempts,
            create_failures=gcp.counts_create_failures,
            rss_bytes=_rss_bytes(),
        )

    # -- reporting -----------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        with self._samples_lock:
            samples = [s.__dict__ for s in self._samples]
        return {
            "samples": samples,
            "writer_lock_hold_ms": {
                "reservoir": self._lock_hold.snapshot(),
                "total": self._lock_hold.total,
                "percentiles": _percentiles(self._lock_hold.snapshot()),
            },
            "dashboard_query_ms": {
                "reservoir": self._query_latency.snapshot(),
                "total": self._query_latency.total,
                "percentiles": _percentiles(self._query_latency.snapshot()),
            },
            "rss_start_bytes": self._rss_start,
            "rss_peak_bytes": max((s.rss_bytes for s in self._samples), default=self._rss_start),
            "rpc_ms": {
                name: {
                    "total": res.total,
                    "percentiles": _percentiles(res.snapshot()),
                }
                for name, res in sorted(self._rpc.items())
            },
        }

    def dump_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(self.to_json(), fh, indent=2)

    def peak_active_threads(self) -> int:
        with self._samples_lock:
            return max((s.active_scale_up_threads for s in self._samples), default=0)

    def rss_delta_bytes(self) -> int:
        with self._samples_lock:
            peak = max((s.rss_bytes for s in self._samples), default=self._rss_start)
        return peak - self._rss_start

    def summary_markdown(self, scenario_name: str, duration_s: float) -> str:
        """Render a human-readable summary with P50/P95/P99 at start/peak/end minute."""
        with self._samples_lock:
            samples = list(self._samples)
        if not samples:
            return f"# {scenario_name}\nNo samples collected.\n"

        # Define minute windows: start (first 60s), peak (60s around peak
        # active_scale_up_threads), end (last 60s).
        total_t = samples[-1].t
        peak_s = max(samples, key=lambda s: s.active_scale_up_threads)
        peak_center = peak_s.t

        def _window(center: float, half: float) -> list[_Sample]:
            lo, hi = center - half, center + half
            return [s for s in samples if lo <= s.t <= hi]

        windows = {
            "start": _window(min(30.0, total_t / 2), 30.0),
            "peak": _window(peak_center, 30.0),
            "end": _window(max(total_t - 30.0, total_t / 2), 30.0),
        }

        peak_threads = peak_s.active_scale_up_threads
        peak_attempts = samples[-1].create_attempts
        peak_failures = samples[-1].create_failures

        lock_stats = _percentiles(self._lock_hold.snapshot())
        query_stats = _percentiles(self._query_latency.snapshot())

        lines = [
            f"# {scenario_name}",
            "",
            f"- duration_s (requested): {duration_s:.1f}",
            f"- samples collected: {len(samples)}",
            f"- peak active_scale_up_threads: {peak_threads} (at t={peak_s.t:.1f}s)",
            f"- total create_attempts: {peak_attempts}",
            f"- total create_failures: {peak_failures}",
            f"- RSS delta: {self.rss_delta_bytes() / 1e6:.1f} MB "
            f"(start={self._rss_start / 1e6:.1f} MB, peak={max(s.rss_bytes for s in samples) / 1e6:.1f} MB)",
            "",
            "## writer_lock_hold_ms (across all writer acquisitions)",
            f"- total acquisitions: {self._lock_hold.total}",
            f"- P50/P95/P99/max (ms): {lock_stats['p50']:.2f} / {lock_stats['p95']:.2f} / "
            f"{lock_stats['p99']:.2f} / {lock_stats['max']:.2f}",
            "",
            "## dashboard_query_ms (probe @2 Hz: SELECT state, count(*) FROM tasks GROUP BY state)",
            f"- total samples: {query_stats['count']}",
            f"- P50/P95/P99/max (ms): {query_stats['p50']:.2f} / {query_stats['p95']:.2f} / "
            f"{query_stats['p99']:.2f} / {query_stats['max']:.2f}",
            "",
            "## rpc_ms (batch wall-clock per WorkerProvider call)",
            "| method | calls | P50 | P95 | P99 | max |",
            "| ------ | ----- | --- | --- | --- | --- |",
        ]
        for name, res in sorted(self._rpc.items()):
            if res.total == 0:
                continue
            p = _percentiles(res.snapshot())
            lines.append(f"| {name} | {res.total} | {p['p50']:.1f} | {p['p95']:.1f} | {p['p99']:.1f} | {p['max']:.1f} |")
        lines += [
            "",
            "## Per-window thread counts (min/mean/max active_scale_up_threads)",
            "| window | samples | min | mean | max |",
            "| ------ | ------- | --- | ---- | --- |",
        ]
        for label, win in windows.items():
            if not win:
                lines.append(f"| {label} | 0 | - | - | - |")
                continue
            vals = [w.active_scale_up_threads for w in win]
            lines.append(f"| {label} | {len(win)} | {min(vals)} | {statistics.fmean(vals):.1f} | {max(vals)} |")
        lines.append("")
        return "\n".join(lines)
