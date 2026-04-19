# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composable RPC probes for the loadtest harness.

Each probe is a small ``(client, state, rng) -> None`` callable that issues a
real RPC through :class:`ControllerServiceClientSync`. :func:`run_probes` spins
one dedicated thread per :class:`ProbeSpec`, each with its own client and its
own fire rate. Latencies are recorded under a ``"<name>@<hz>hz"`` key so the
same RPC polled at two different rates is distinguishable in the report.

The probes intentionally do *not* consult the controller's DB directly — they
go through the RPC server like a real dashboard client, so they exercise the
same serialization, auth, and thread-pool paths as production.
"""

from __future__ import annotations

import logging
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from iris.rpc import controller_pb2, query_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync

logger = logging.getLogger(__name__)


@dataclass
class ClusterState:
    """Snapshot the harness exposes to probes at startup.

    ``pending_job_ids`` / ``running_job_ids`` are picked at random per call by
    the ``get_job_status_*`` probes. When empty, the corresponding probe
    no-ops for that tick rather than raising.
    """

    controller_url: str
    pending_job_ids: list[str] = field(default_factory=list)
    running_job_ids: list[str] = field(default_factory=list)
    rng_seed: int = 0


@dataclass(frozen=True)
class ProbeSpec:
    """One probe at one fire rate. The same ``name`` may appear multiple times."""

    name: str
    hz: float

    def key(self) -> str:
        return f"{self.name}@{self.hz}hz"


class ProbeCallable(Protocol):
    def __call__(self, client: ControllerServiceClientSync, state: ClusterState, rng: random.Random) -> None: ...


@dataclass
class ProbeResult:
    """Latency samples keyed by ``ProbeSpec.key()``."""

    latencies_ms: dict[str, list[float]] = field(default_factory=dict)

    def add(self, key: str, elapsed_ms: float) -> None:
        self.latencies_ms.setdefault(key, []).append(elapsed_ms)

    def percentiles(self, key: str) -> dict[str, float]:
        samples = sorted(self.latencies_ms.get(key, []))
        if not samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0, "max": 0.0}
        n = len(samples)

        def q(p: float) -> float:
            idx = min(n - 1, max(0, round(p * (n - 1))))
            return samples[idx]

        return {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "count": n, "max": samples[-1]}


# ---------------------------------------------------------------------------
# Probe implementations — all take (client, state, rng) and raise on failure.
# The runner records latency regardless of exceptions.
# ---------------------------------------------------------------------------


def _probe_list_jobs(client: ControllerServiceClientSync, state: ClusterState, rng: random.Random) -> None:
    del state, rng
    req = controller_pb2.Controller.ListJobsRequest()
    # ListJobsRequest wraps query parameters in a nested JobQuery message; the
    # flat ``.limit`` attribute does not exist and using it raises.
    req.query.limit = 50
    client.list_jobs(req)


def _probe_get_job_status_pending(client: ControllerServiceClientSync, state: ClusterState, rng: random.Random) -> None:
    if not state.pending_job_ids:
        return
    job_id = rng.choice(state.pending_job_ids)
    client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=job_id))


def _probe_get_job_status_running(client: ControllerServiceClientSync, state: ClusterState, rng: random.Random) -> None:
    if not state.running_job_ids:
        return
    job_id = rng.choice(state.running_job_ids)
    client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=job_id))


def _probe_list_workers(client: ControllerServiceClientSync, state: ClusterState, rng: random.Random) -> None:
    del state, rng
    client.list_workers(controller_pb2.Controller.ListWorkersRequest())


def _probe_scheduler_state(client: ControllerServiceClientSync, state: ClusterState, rng: random.Random) -> None:
    del state, rng
    client.get_scheduler_state(controller_pb2.Controller.GetSchedulerStateRequest())


def _probe_list_endpoints(client: ControllerServiceClientSync, state: ClusterState, rng: random.Random) -> None:
    del state, rng
    client.list_endpoints(controller_pb2.Controller.ListEndpointsRequest(prefix=""))


def _probe_execute_raw_query(client: ControllerServiceClientSync, state: ClusterState, rng: random.Random) -> None:
    del state, rng
    client.execute_raw_query(query_pb2.RawQueryRequest(sql="SELECT state, COUNT(*) FROM tasks GROUP BY state"))


PROBES: dict[str, ProbeCallable] = {
    "list_jobs": _probe_list_jobs,
    "get_job_status_pending": _probe_get_job_status_pending,
    "get_job_status_running": _probe_get_job_status_running,
    "list_workers": _probe_list_workers,
    "scheduler_state": _probe_scheduler_state,
    "list_endpoints": _probe_list_endpoints,
    "execute_raw_query": _probe_execute_raw_query,
}


# ---------------------------------------------------------------------------
# Spec parsing + runner
# ---------------------------------------------------------------------------


def parse_probe_spec(token: str) -> ProbeSpec:
    """Parse ``name:hz`` into a :class:`ProbeSpec`. Raises on unknown probe names."""
    parts = token.split(":")
    if len(parts) != 2:
        raise ValueError(f"probe spec must be 'name:hz', got {token!r}")
    name, hz_str = parts[0].strip(), parts[1].strip()
    if name not in PROBES:
        raise ValueError(f"unknown probe {name!r}; available: {sorted(PROBES)}")
    try:
        hz = float(hz_str)
    except ValueError as exc:
        raise ValueError(f"probe hz must be a float, got {hz_str!r}") from exc
    if hz <= 0:
        raise ValueError(f"probe hz must be > 0, got {hz}")
    return ProbeSpec(name=name, hz=hz)


def parse_probe_specs(tokens: list[str]) -> list[ProbeSpec]:
    return [parse_probe_spec(t) for t in tokens]


def _run_one_spec(
    spec: ProbeSpec,
    state: ClusterState,
    result: ProbeResult,
    result_lock: threading.Lock,
    stop: threading.Event,
    deadline: float,
    timeout_ms: int,
) -> None:
    probe = PROBES[spec.name]
    rng = random.Random(hash((state.rng_seed, spec.name, spec.hz)) & 0xFFFFFFFF)
    client = ControllerServiceClientSync(address=state.controller_url, timeout_ms=timeout_ms)
    interval = 1.0 / spec.hz
    key = spec.key()
    try:
        while not stop.is_set() and time.monotonic() < deadline:
            t0 = time.monotonic()
            try:
                probe(client, state, rng)
            except Exception:
                logger.debug("probe %s failed", key, exc_info=True)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            with result_lock:
                result.add(key, elapsed_ms)
            if stop.wait(interval):
                return
    finally:
        try:
            client.close()
        except Exception:
            pass


def run_probes(
    state: ClusterState,
    specs: list[ProbeSpec],
    *,
    duration_seconds: float,
    stop: threading.Event | None = None,
    timeout_ms: int = 15_000,
) -> ProbeResult:
    """Run each :class:`ProbeSpec` on its own thread for ``duration_seconds``.

    Each thread opens its own :class:`ControllerServiceClientSync`. Latencies
    are recorded regardless of exception — a 15 s timeout is the default so
    genuinely slow RPCs are measured, not clipped.
    """
    if not specs:
        raise ValueError("run_probes: specs list is empty")

    result = ProbeResult()
    result_lock = threading.Lock()
    stop = stop or threading.Event()
    deadline = time.monotonic() + duration_seconds

    threads: list[threading.Thread] = []
    for i, spec in enumerate(specs):
        t = threading.Thread(
            target=_run_one_spec,
            name=f"probe-{i}-{spec.key()}",
            args=(spec, state, result, result_lock, stop, deadline, timeout_ms),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=duration_seconds + 10.0)
    return result


# ---------------------------------------------------------------------------
# Snapshot helper: picks pending/running job ids the probes pull from.
# ---------------------------------------------------------------------------


def load_snapshot_pending_running_job_ids(snapshot_db: Path | str, limit: int = 500) -> tuple[list[str], list[str]]:
    """Return ``(pending, running)`` job id lists from the snapshot DB.

    Opens the sqlite file read-only so this is safe to call while the harness
    mutates its own working copy.
    """
    uri = f"file:{snapshot_db}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        # job_pb2 values: PENDING=1, BUILDING=2, RUNNING=3.
        pending = [
            r[0]
            for r in conn.execute(
                "SELECT job_id FROM jobs WHERE state = 1 ORDER BY submitted_at_ms DESC LIMIT ?", (limit,)
            )
        ]
        running = [
            r[0]
            for r in conn.execute(
                "SELECT job_id FROM jobs WHERE state = 3 ORDER BY submitted_at_ms DESC LIMIT ?", (limit,)
            )
        ]
    finally:
        conn.close()
    return pending, running


def load_live_pending_running_job_ids(db, limit: int = 500) -> tuple[list[str], list[str]]:
    """Read pending/running job ids from the harness's live ControllerDB."""
    with db.read_snapshot() as q:
        pending = [
            r[0]
            for r in q.execute_sql(
                "SELECT job_id FROM jobs WHERE state = 1 ORDER BY submitted_at_ms DESC LIMIT ?", (limit,)
            ).fetchall()
        ]
        running = [
            r[0]
            for r in q.execute_sql(
                "SELECT job_id FROM jobs WHERE state = 3 ORDER BY submitted_at_ms DESC LIMIT ?", (limit,)
            ).fetchall()
        ]
    return pending, running


# ---------------------------------------------------------------------------
# Default probe mixes (used by scenarios).
# ---------------------------------------------------------------------------


# Stage-10c mix with running-job polling added. Running jobs get polled more
# often than pending ones — that's the per-user behavior (watching an in-flight
# job's progress page).
INCIDENT_PROBES: list[ProbeSpec] = [
    ProbeSpec("list_jobs", hz=1.0),  # dashboard tab
    ProbeSpec("list_jobs", hz=5.0),  # job-page sidebar
    ProbeSpec("get_job_status_running", hz=5.0),  # running job polling
    ProbeSpec("get_job_status_pending", hz=1.0),
    ProbeSpec("list_workers", hz=0.1),
    ProbeSpec("scheduler_state", hz=0.1),
]


# Prod-magnitude mix: the incident mix plus endpoint + raw-query probes.
FLEET_WIDE_PROBES: list[ProbeSpec] = [
    *INCIDENT_PROBES,
    ProbeSpec("list_endpoints", hz=0.5),
    ProbeSpec("execute_raw_query", hz=0.1),
]


# Aliases used as scenario defaults:
BURST_PROBES: list[ProbeSpec] = INCIDENT_PROBES
API_TIMEOUTS_PROBES: list[ProbeSpec] = INCIDENT_PROBES
PROD_SCALE_PROBES: list[ProbeSpec] = FLEET_WIDE_PROBES


__all__ = [
    "API_TIMEOUTS_PROBES",
    "BURST_PROBES",
    "FLEET_WIDE_PROBES",
    "INCIDENT_PROBES",
    "PROBES",
    "PROD_SCALE_PROBES",
    "ClusterState",
    "ProbeCallable",
    "ProbeResult",
    "ProbeSpec",
    "load_live_pending_running_job_ids",
    "load_snapshot_pending_running_job_ids",
    "parse_probe_spec",
    "parse_probe_specs",
    "run_probes",
]
